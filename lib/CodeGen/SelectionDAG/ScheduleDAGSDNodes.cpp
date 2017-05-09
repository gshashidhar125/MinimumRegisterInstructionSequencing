//===--- ScheduleDAGSDNodes.cpp - Implement the ScheduleDAGSDNodes class --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAG class, which is a base class used by
// scheduling implementation classes.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "SDNodeDbgValue.h"
#include "ScheduleDAGSDNodes.h"
#include "InstrEmitter.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
#include <stack>
#include <bitset>
#include <vector>
using namespace llvm;

STATISTIC(LoadsClustered, "Number of loads clustered together");

// This allows latency based scheduler to notice high latency instructions
// without a target itinerary. The choise if number here has more to do with
// balancing scheduler heursitics than with the actual machine latency.
static cl::opt<int> HighLatencyCycles(
  "sched-high-latency-cycles", cl::Hidden, cl::init(10),
  cl::desc("Roughly estimate the number of cycles that 'long latency'"
           "instructions take for targets with no itinerary"));

ScheduleDAGSDNodes::ScheduleDAGSDNodes(MachineFunction &mf)
  : ScheduleDAG(mf),
    InstrItins(mf.getTarget().getInstrItineraryData()) {}

/// Run - perform scheduling.
///
void ScheduleDAGSDNodes::Run(SelectionDAG *dag, MachineBasicBlock *bb,
                             MachineBasicBlock::iterator insertPos) {
  DAG = dag;
  ScheduleDAG::Run(bb, insertPos);
}

/// NewSUnit - Creates a new SUnit and return a ptr to it.
///
SUnit *ScheduleDAGSDNodes::NewSUnit(SDNode *N) {
#ifndef NDEBUG
  const SUnit *Addr = 0;
  if (!SUnits.empty())
    Addr = &SUnits[0];
#endif
  SUnits.push_back(SUnit(N, (unsigned)SUnits.size()));
  assert((Addr == 0 || Addr == &SUnits[0]) &&
         "SUnits std::vector reallocated on the fly!");
  SUnits.back().OrigNode = &SUnits.back();
  SUnit *SU = &SUnits.back();
  const TargetLowering &TLI = DAG->getTargetLoweringInfo();
  if (!N ||
      (N->isMachineOpcode() &&
       N->getMachineOpcode() == TargetOpcode::IMPLICIT_DEF))
    SU->SchedulingPref = Sched::None;
  else
    SU->SchedulingPref = TLI.getSchedulingPreference(N);
  return SU;
}

SUnit *ScheduleDAGSDNodes::Clone(SUnit *Old) {
  SUnit *SU = NewSUnit(Old->getNode());
  SU->OrigNode = Old->OrigNode;
  SU->Latency = Old->Latency;
  SU->isVRegCycle = Old->isVRegCycle;
  SU->isCall = Old->isCall;
  SU->isCallOp = Old->isCallOp;
  SU->isTwoAddress = Old->isTwoAddress;
  SU->isCommutable = Old->isCommutable;
  SU->hasPhysRegDefs = Old->hasPhysRegDefs;
  SU->hasPhysRegClobbers = Old->hasPhysRegClobbers;
  SU->isScheduleHigh = Old->isScheduleHigh;
  SU->isScheduleLow = Old->isScheduleLow;
  SU->SchedulingPref = Old->SchedulingPref;
  Old->isCloned = true;
  return SU;
}

/// CheckForPhysRegDependency - Check if the dependency between def and use of
/// a specified operand is a physical register dependency. If so, returns the
/// register and the cost of copying the register.
static void CheckForPhysRegDependency(SDNode *Def, SDNode *User, unsigned Op,
                                      const TargetRegisterInfo *TRI,
                                      const TargetInstrInfo *TII,
                                      unsigned &PhysReg, int &Cost) {
  if (Op != 2 || User->getOpcode() != ISD::CopyToReg)
    return;

  unsigned Reg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
  if (TargetRegisterInfo::isVirtualRegister(Reg))
    return;

  unsigned ResNo = User->getOperand(2).getResNo();
  if (Def->isMachineOpcode()) {
    const MCInstrDesc &II = TII->get(Def->getMachineOpcode());
    if (ResNo >= II.getNumDefs() &&
        II.ImplicitDefs[ResNo - II.getNumDefs()] == Reg) {
      PhysReg = Reg;
      const TargetRegisterClass *RC =
        TRI->getMinimalPhysRegClass(Reg, Def->getValueType(ResNo));
      Cost = RC->getCopyCost();
    }
  }
}

static void AddGlue(SDNode *N, SDValue Glue, bool AddGlue, SelectionDAG *DAG) {
  SmallVector<EVT, 4> VTs;
  SDNode *GlueDestNode = Glue.getNode();

  // Don't add glue from a node to itself.
  if (GlueDestNode == N) return;

  // Don't add glue to something which already has glue.
  if (N->getValueType(N->getNumValues() - 1) == MVT::Glue) return;

  for (unsigned I = 0, E = N->getNumValues(); I != E; ++I)
    VTs.push_back(N->getValueType(I));

  if (AddGlue)
    VTs.push_back(MVT::Glue);

  SmallVector<SDValue, 4> Ops;
  for (unsigned I = 0, E = N->getNumOperands(); I != E; ++I)
    Ops.push_back(N->getOperand(I));

  if (GlueDestNode)
    Ops.push_back(Glue);

  SDVTList VTList = DAG->getVTList(&VTs[0], VTs.size());
  MachineSDNode::mmo_iterator Begin = 0, End = 0;
  MachineSDNode *MN = dyn_cast<MachineSDNode>(N);

  // Store memory references.
  if (MN) {
    Begin = MN->memoperands_begin();
    End = MN->memoperands_end();
  }

  DAG->MorphNodeTo(N, N->getOpcode(), VTList, &Ops[0], Ops.size());

  // Reset the memory references
  if (MN)
    MN->setMemRefs(Begin, End);
}

/// ClusterNeighboringLoads - Force nearby loads together by "gluing" them.
/// This function finds loads of the same base and different offsets. If the
/// offsets are not far apart (target specific), it add MVT::Glue inputs and
/// outputs to ensure they are scheduled together and in order. This
/// optimization may benefit some targets by improving cache locality.
void ScheduleDAGSDNodes::ClusterNeighboringLoads(SDNode *Node) {
  SDNode *Chain = 0;
  unsigned NumOps = Node->getNumOperands();
  if (Node->getOperand(NumOps-1).getValueType() == MVT::Other)
    Chain = Node->getOperand(NumOps-1).getNode();
  if (!Chain)
    return;

  // Look for other loads of the same chain. Find loads that are loading from
  // the same base pointer and different offsets.
  SmallPtrSet<SDNode*, 16> Visited;
  SmallVector<int64_t, 4> Offsets;
  DenseMap<long long, SDNode*> O2SMap;  // Map from offset to SDNode.
  bool Cluster = false;
  SDNode *Base = Node;
  for (SDNode::use_iterator I = Chain->use_begin(), E = Chain->use_end();
       I != E; ++I) {
    SDNode *User = *I;
    if (User == Node || !Visited.insert(User))
      continue;
    int64_t Offset1, Offset2;
    if (!TII->areLoadsFromSameBasePtr(Base, User, Offset1, Offset2) ||
        Offset1 == Offset2)
      // FIXME: Should be ok if they addresses are identical. But earlier
      // optimizations really should have eliminated one of the loads.
      continue;
    if (O2SMap.insert(std::make_pair(Offset1, Base)).second)
      Offsets.push_back(Offset1);
    O2SMap.insert(std::make_pair(Offset2, User));
    Offsets.push_back(Offset2);
    if (Offset2 < Offset1)
      Base = User;
    Cluster = true;
  }

  if (!Cluster)
    return;

  // Sort them in increasing order.
  std::sort(Offsets.begin(), Offsets.end());

  // Check if the loads are close enough.
  SmallVector<SDNode*, 4> Loads;
  unsigned NumLoads = 0;
  int64_t BaseOff = Offsets[0];
  SDNode *BaseLoad = O2SMap[BaseOff];
  Loads.push_back(BaseLoad);
  for (unsigned i = 1, e = Offsets.size(); i != e; ++i) {
    int64_t Offset = Offsets[i];
    SDNode *Load = O2SMap[Offset];
    if (!TII->shouldScheduleLoadsNear(BaseLoad, Load, BaseOff, Offset,NumLoads))
      break; // Stop right here. Ignore loads that are further away.
    Loads.push_back(Load);
    ++NumLoads;
  }

  if (NumLoads == 0)
    return;

  // Cluster loads by adding MVT::Glue outputs and inputs. This also
  // ensure they are scheduled in order of increasing addresses.
  SDNode *Lead = Loads[0];
  AddGlue(Lead, SDValue(0, 0), true, DAG);

  SDValue InGlue = SDValue(Lead, Lead->getNumValues() - 1);
  for (unsigned I = 1, E = Loads.size(); I != E; ++I) {
    bool OutGlue = I < E - 1;
    SDNode *Load = Loads[I];

    AddGlue(Load, InGlue, OutGlue, DAG);

    if (OutGlue)
      InGlue = SDValue(Load, Load->getNumValues() - 1);

    ++LoadsClustered;
  }
}

/// ClusterNodes - Cluster certain nodes which should be scheduled together.
///
void ScheduleDAGSDNodes::ClusterNodes() {
  for (SelectionDAG::allnodes_iterator NI = DAG->allnodes_begin(),
       E = DAG->allnodes_end(); NI != E; ++NI) {
    SDNode *Node = &*NI;
    if (!Node || !Node->isMachineOpcode())
      continue;

    unsigned Opc = Node->getMachineOpcode();
    const MCInstrDesc &MCID = TII->get(Opc);
    if (MCID.mayLoad())
      // Cluster loads from "near" addresses into combined SUnits.
      ClusterNeighboringLoads(Node);
  }
}

void ScheduleDAGSDNodes::BuildSchedUnits() {
  // During scheduling, the NodeId field of SDNode is used to map SDNodes
  // to their associated SUnits by holding SUnits table indices. A value
  // of -1 means the SDNode does not yet have an associated SUnit.
  unsigned NumNodes = 0;
  for (SelectionDAG::allnodes_iterator NI = DAG->allnodes_begin(),
       E = DAG->allnodes_end(); NI != E; ++NI) {
    NI->setNodeId(-1);
    ++NumNodes;
  }

  // Reserve entries in the vector for each of the SUnits we are creating.  This
  // ensure that reallocation of the vector won't happen, so SUnit*'s won't get
  // invalidated.
  // FIXME: Multiply by 2 because we may clone nodes during scheduling.
  // This is a temporary workaround.
  SUnits.reserve(NumNodes * 2);

  // Add all nodes in depth first order.
  SmallVector<SDNode*, 64> Worklist;
  SmallPtrSet<SDNode*, 64> Visited;
  Worklist.push_back(DAG->getRoot().getNode());
  Visited.insert(DAG->getRoot().getNode());

  SmallVector<SUnit*, 8> CallSUnits;
  while (!Worklist.empty()) {
    SDNode *NI = Worklist.pop_back_val();

    // Add all operands to the worklist unless they've already been added.
    for (unsigned i = 0, e = NI->getNumOperands(); i != e; ++i)
      if (Visited.insert(NI->getOperand(i).getNode()))
        Worklist.push_back(NI->getOperand(i).getNode());

    if (isPassiveNode(NI))  // Leaf node, e.g. a TargetImmediate.
      continue;

    // If this node has already been processed, stop now.
    if (NI->getNodeId() != -1) continue;

    SUnit *NodeSUnit = NewSUnit(NI);

    // See if anything is glued to this node, if so, add them to glued
    // nodes.  Nodes can have at most one glue input and one glue output.  Glue
    // is required to be the last operand and result of a node.

    // Scan up to find glued preds.
    SDNode *N = NI;
    while (N->getNumOperands() &&
           N->getOperand(N->getNumOperands()-1).getValueType() == MVT::Glue) {
      N = N->getOperand(N->getNumOperands()-1).getNode();
      assert(N->getNodeId() == -1 && "Node already inserted!");
      N->setNodeId(NodeSUnit->NodeNum);
      if (N->isMachineOpcode() && TII->get(N->getMachineOpcode()).isCall())
        NodeSUnit->isCall = true;
    }

    // Scan down to find any glued succs.
    N = NI;
    while (N->getValueType(N->getNumValues()-1) == MVT::Glue) {
      SDValue GlueVal(N, N->getNumValues()-1);

      // There are either zero or one users of the Glue result.
      bool HasGlueUse = false;
      for (SDNode::use_iterator UI = N->use_begin(), E = N->use_end();
           UI != E; ++UI)
        if (GlueVal.isOperandOf(*UI)) {
          HasGlueUse = true;
          assert(N->getNodeId() == -1 && "Node already inserted!");
          N->setNodeId(NodeSUnit->NodeNum);
          N = *UI;
          if (N->isMachineOpcode() && TII->get(N->getMachineOpcode()).isCall())
            NodeSUnit->isCall = true;
          break;
        }
      if (!HasGlueUse) break;
    }

    if (NodeSUnit->isCall)
      CallSUnits.push_back(NodeSUnit);

    // Schedule zero-latency TokenFactor below any nodes that may increase the
    // schedule height. Otherwise, ancestors of the TokenFactor may appear to
    // have false stalls.
    if (NI->getOpcode() == ISD::TokenFactor)
      NodeSUnit->isScheduleLow = true;

    // If there are glue operands involved, N is now the bottom-most node
    // of the sequence of nodes that are glued together.
    // Update the SUnit.
    NodeSUnit->setNode(N);
    assert(N->getNodeId() == -1 && "Node already inserted!");
    N->setNodeId(NodeSUnit->NodeNum);

    // Compute NumRegDefsLeft. This must be done before AddSchedEdges.
    InitNumRegDefsLeft(NodeSUnit);

    // Assign the Latency field of NodeSUnit using target-provided information.
    ComputeLatency(NodeSUnit);
  }

  // Find all call operands.
  while (!CallSUnits.empty()) {
    SUnit *SU = CallSUnits.pop_back_val();
    for (const SDNode *SUNode = SU->getNode(); SUNode;
         SUNode = SUNode->getGluedNode()) {
      if (SUNode->getOpcode() != ISD::CopyToReg)
        continue;
      SDNode *SrcN = SUNode->getOperand(2).getNode();
      if (isPassiveNode(SrcN)) continue;   // Not scheduled.
      SUnit *SrcSU = &SUnits[SrcN->getNodeId()];
      SrcSU->isCallOp = true;
    }
  }
}

void ScheduleDAGSDNodes::AddSchedEdges() {
  const TargetSubtargetInfo &ST = TM.getSubtarget<TargetSubtargetInfo>();

  // Check to see if the scheduler cares about latencies.
  bool UnitLatencies = ForceUnitLatencies();

  // Pass 2: add the preds, succs, etc.
  for (unsigned su = 0, e = SUnits.size(); su != e; ++su) {
    SUnit *SU = &SUnits[su];
    SDNode *MainNode = SU->getNode();

    if (MainNode->isMachineOpcode()) {
      unsigned Opc = MainNode->getMachineOpcode();
      const MCInstrDesc &MCID = TII->get(Opc);
      for (unsigned i = 0; i != MCID.getNumOperands(); ++i) {
        if (MCID.getOperandConstraint(i, MCOI::TIED_TO) != -1) {
          SU->isTwoAddress = true;
          break;
        }
      }
      if (MCID.isCommutable())
        SU->isCommutable = true;
    }

    // Find all predecessors and successors of the group.
    for (SDNode *N = SU->getNode(); N; N = N->getGluedNode()) {
      if (N->isMachineOpcode() &&
          TII->get(N->getMachineOpcode()).getImplicitDefs()) {
        SU->hasPhysRegClobbers = true;
        unsigned NumUsed = InstrEmitter::CountResults(N);
        while (NumUsed != 0 && !N->hasAnyUseOfValue(NumUsed - 1))
          --NumUsed;    // Skip over unused values at the end.
        if (NumUsed > TII->get(N->getMachineOpcode()).getNumDefs())
          SU->hasPhysRegDefs = true;
      }

      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
        SDNode *OpN = N->getOperand(i).getNode();
        if (isPassiveNode(OpN)) continue;   // Not scheduled.
        SUnit *OpSU = &SUnits[OpN->getNodeId()];
        assert(OpSU && "Node has no SUnit!");
        if (OpSU == SU) continue;           // In the same group.

        EVT OpVT = N->getOperand(i).getValueType();
        assert(OpVT != MVT::Glue && "Glued nodes should be in same sunit!");
        bool isChain = OpVT == MVT::Other;

        unsigned PhysReg = 0;
        int Cost = 1;
        // Determine if this is a physical register dependency.
        CheckForPhysRegDependency(OpN, N, i, TRI, TII, PhysReg, Cost);
        assert((PhysReg == 0 || !isChain) &&
               "Chain dependence via physreg data?");
        // FIXME: See ScheduleDAGSDNodes::EmitCopyFromReg. For now, scheduler
        // emits a copy from the physical register to a virtual register unless
        // it requires a cross class copy (cost < 0). That means we are only
        // treating "expensive to copy" register dependency as physical register
        // dependency. This may change in the future though.
        if (Cost >= 0 && !StressSched)
          PhysReg = 0;

        // If this is a ctrl dep, latency is 1.
        unsigned OpLatency = isChain ? 1 : OpSU->Latency;
        // Special-case TokenFactor chains as zero-latency.
        if(isChain && OpN->getOpcode() == ISD::TokenFactor)
          OpLatency = 0;

        const SDep &dep = SDep(OpSU, isChain ? SDep::Order : SDep::Data,
                               OpLatency, PhysReg);
        if (!isChain && !UnitLatencies) {
          ComputeOperandLatency(OpN, N, i, const_cast<SDep &>(dep));
          ST.adjustSchedDependency(OpSU, SU, const_cast<SDep &>(dep));
        }

        if (!SU->addPred(dep) && !dep.isCtrl() && OpSU->NumRegDefsLeft > 1) {
          // Multiple register uses are combined in the same SUnit. For example,
          // we could have a set of glued nodes with all their defs consumed by
          // another set of glued nodes. Register pressure tracking sees this as
          // a single use, so to keep pressure balanced we reduce the defs.
          //
          // We can't tell (without more book-keeping) if this results from
          // glued nodes or duplicate operands. As long as we don't reduce
          // NumRegDefsLeft to zero, we handle the common cases well.
          --OpSU->NumRegDefsLeft;
        }
      }
    }
  }
}

void ScheduleDAGSDNodes::ComputeHeight(std::vector<int> &Height) {

	SUnit *temp;
	unsigned NumberOfSUnits = SUnits.size(), i;
	std::vector<int> SUnitSuccsLeft;
	std::vector<SUnit*> WorkList;

	SUnitSuccsLeft.resize(NumberOfSUnits);
	Height.resize(NumberOfSUnits);
	WorkList.reserve(NumberOfSUnits);
	
	for (i = 0; i < NumberOfSUnits; i++) {

		temp = &SUnits[i];
		SUnitSuccsLeft[temp->NodeNum] = temp->Succs.size();

		if (SUnitSuccsLeft[temp->NodeNum] == 0) {

			Height[temp->NodeNum] = 1;
			WorkList.push_back(temp);
		}
	}

//	dbgs() << "\n\nHeight Calculation \n";
	while (!WorkList.empty()) {

		int MaxHeight = 0;
		temp = WorkList.back();
		WorkList.pop_back();
		for (SUnit::succ_iterator I = temp->Succs.begin(), E = temp->Succs.end();I != E; ++I) {

			if (MaxHeight < Height[I->getSUnit()->NodeNum])
				MaxHeight = Height[I->getSUnit()->NodeNum];
		}
		Height[temp->NodeNum] = MaxHeight + 1;
		temp->LineageHeight = Height[temp->NodeNum];
//		dbgs() << "Sunit" << temp->NodeNum << "Height= " << Height[temp->NodeNum] << " \n";
		for (SUnit::pred_iterator I = temp->Preds.begin(), E = temp->Preds.end(); I != E; ++I) {

			SUnit *SU = I->getSUnit();
			if (!--SUnitSuccsLeft[SU->NodeNum])
				WorkList.push_back(SU);
		}
	}
//	dbgs() << "\n\n";
}

struct CompareHeightOfLineage {
	bool operator()(SUnit *A, SUnit *B) const {
		return A->LineageHeight <= B->LineageHeight;
	}
};

void DumpSUnits(std::vector<SUnit> SUnits) {

	unsigned i = 0;
	SUnit *temp;

	for (i = 0; i < SUnits.size(); i++) {

		temp = &SUnits[i];
		dbgs() << "Sunit" << i << ", Pointer=" << temp  << ", SDNode=" << temp->getNode() << ",SUnit OrigNode:" << temp->OrigNode << ",NodeNum:" << temp->NodeNum << ",NumPred:" << temp->NumPreds << ",NumSuccs:" << temp->NumSuccs << ",Latency:" << temp->Latency << ",Pref:" << temp->SchedulingPref << ",Depth:" << temp->getDepth() << ",Height:" << temp->getHeight() << " \n";

		for (SUnit::pred_iterator PI = temp->Preds.begin(), PE = temp->Preds.end() ; PI != PE; PI++) {

			dbgs() << "\tPredecessor: " << PI << ", SDNode= " << PI->getSUnit() << ", Kind= " << PI->getKind() << "\n";
		}
		dbgs() << "\n";
		for (SUnit::succ_iterator SI = temp->Succs.begin(), SE = temp->Succs.end() ; SI != SE; SI++) {

			dbgs() << "\tSuccessor: " << SI << ", SDNode= " << SI->getSUnit() << ", Kind= " << SI->getKind() << "\n";
		}
	}
}

void ScheduleDAGSDNodes::BuildDDG() {

	//DumpSUnits(SUnits);

	SUnit *temp, *Heir;
	unsigned i = 0, NumberOfSUnits = SUnits.size();
	int lineageNumber = 0, LowerDescendantHeight = 0;
	Lineage *tempLineage;
	std::vector<int> Height;
	std::vector<bool>InLineage;
	bool RecomputeHeight = false;
	std::priority_queue<SUnit *, std::vector<SUnit *>, CompareHeightOfLineage> HeightOfSUnits;
	
	InLineage.resize(NumberOfSUnits);

	ComputeHeight(Height);

	for (i = 0; i < NumberOfSUnits; i++) {

		temp = &SUnits[i];
		InLineage[i] = false;
		HeightOfSUnits.push(temp);
	}
	
	while (!HeightOfSUnits.empty()) {

		while (!HeightOfSUnits.empty()) {
			temp = HeightOfSUnits.top();
			HeightOfSUnits.pop();
			if (InLineage[temp->NodeNum] == false)
				break;
		}

		Heir = NULL;
		RecomputeHeight = false;

		if (temp->NumSuccs != 0) {

			tempLineage = new Lineage(lineageNumber++);
			tempLineage->setNode(temp);
			tempLineage->StartNode = temp;
			temp->lineage = tempLineage;
			ListOfLineage.push_back(tempLineage);
			InLineage[temp->NodeNum] = true;
			//dbgs() << "New Lineage: " << tempLineage->Id << "\n";
		}
		else
			continue;

		while (temp->NumSuccs != 0) {

			LowerDescendantHeight = 0;
			//dbgs() << "\tSUnit: " << temp->NodeNum << " , Height= " << Height[temp->NodeNum] << " \n";
			for (SUnit::succ_iterator SI = temp->Succs.begin(), SE = temp->Succs.end() ; SI != SE; SI++) {

				if (SI->getKind() == SDep::Data) {

					if (!LowerDescendantHeight || (LowerDescendantHeight > Height[SI->getSUnit()->NodeNum])) {

						Heir = SI->getSUnit();
						LowerDescendantHeight = Height[SI->getSUnit()->NodeNum];
					}
				}
			}

			if (Heir) {

				tempLineage->setNode(Heir);
				//dbgs() << "\tHeir: " << Heir->NodeNum << " , Height= " << Heir->LineageHeight << ", Lineage: " << tempLineage->Id << "\n";
			}

			for (SUnit::succ_iterator SI = temp->Succs.begin(), SE = temp->Succs.end() ; (Heir != NULL) && SI != SE; SI++) {

				if (SI->getKind() == SDep::Data && (SI->getSUnit() != Heir)) {

					//dbgs() << "\t\tSiblings: " << SI->getSUnit()->NodeNum << " , Height= " << SI->getSUnit()->LineageHeight << " \n";
					RecomputeHeight = true;
					SDep SequencingEdge(SI->getSUnit(), SDep::Order);
					Heir->addPred(SequencingEdge);
				}
			}
			if (InLineage[Heir->NodeNum] == true) {
				tempLineage->EndNode = Heir;
				//dbgs() << "\tEnd of Lineage: " << tempLineage->Id <<  " \n";
				break;
			}
			Heir->lineage = tempLineage;
			InLineage[Heir->NodeNum] = true;
			if (Heir->NumSuccs == 0)
				tempLineage->EndNode = Heir;
			temp = Heir;
		}
		if (RecomputeHeight) {
			ComputeHeight(Height);
		}
	}

	dbgs() << "\n\nList of Lineages\n";
	unsigned NumberOfLineages = ListOfLineage.size();
	for (i = 0; i < ListOfLineage.size(); i++) {

		Lineage *L = ListOfLineage[i];
		dbgs() << "Lineage: " << L->Id << "StartNode: " << L->StartNode->NodeNum << ", EndNode: " << L->EndNode->NodeNum << ", List of Nodes: ";
		for (Lineage::Nodes_iterator LI = L->Nodes.begin(), LE = L->Nodes.end(); LI != LE; LI++) {

			dbgs() << (*LI)->NodeNum << " ";
		}
		dbgs() << "\n";
	}
	dbgs() << "\n\n";

	int Degree, TopoIndex = NumberOfSUnits;
	std::vector<SUnit *> WorkList;
	std::vector<int> SUnitSuccsLeft;
	std::vector<int> SUnitTopoIndex;

	WorkList.clear();
	SUnitSuccsLeft.resize(NumberOfSUnits);
	SUnitTopoIndex.resize(NumberOfSUnits);

	//DumpSUnits(SUnits);

	for (i = 0; i < NumberOfSUnits; i++) {
	
		temp = &SUnits[i];
		Degree = temp->Succs.size();
		SUnitSuccsLeft[temp->NodeNum] = Degree;
		if (Degree == 0) {
			WorkList.push_back(temp);
		}
	}

	while (!WorkList.empty()) {
	
		temp = WorkList.back();
		WorkList.pop_back();
		SUnitTopoIndex[temp->NodeNum] = --TopoIndex;

		for (SUnit::const_pred_iterator PI = temp->Preds.begin(), PE = temp->Preds.end(); PI != PE; PI++) {
		
			SUnit *SU = PI->getSUnit();
			if (!--SUnitSuccsLeft[SU->NodeNum])
				WorkList.push_back(SU);
		}
	}

	for (i = 0; i < NumberOfSUnits; i++) {

		SUnit *SU = &SUnits[i];
		for (SUnit::const_pred_iterator PI = SU->Preds.begin(), PE = SU->Preds.end(); PI != PE; PI++) {
			assert(SUnitTopoIndex[SU->NodeNum] > SUnitTopoIndex[PI->getSUnit()->NodeNum] && "Wrong topological sorting");
		}
	}

/*	dbgs() << "\n\nTopological Ordering\n";
	for (i = 0; i < NumberOfSUnits; i++) {

		temp = &SUnits[i];
		dbgs() << "Topo Ordering["<< temp->NodeNum << "]: " << SUnitTopoIndex[temp->NodeNum] << "\n";
	}
	dbgs() << "\n\n";
*/	
	int graph[1000][1000];
	unsigned j, k;
	for (i = 0; i < 1000; i++)
		for (j = 0; j < 1000; j++)
			graph[i][j] = 0;
	
	for (i = 0; i < NumberOfSUnits; i++) {
		
		temp = &SUnits[i];
		graph[temp->NodeNum][temp->NodeNum] = 1;

		for (SUnit::pred_iterator PI = temp->Preds.begin(), PE = temp->Preds.end() ; PI != PE; PI++)
			graph[PI->getSUnit()->NodeNum][temp->NodeNum] = 1;

		for (SUnit::succ_iterator SI = temp->Succs.begin(), SE = temp->Succs.end() ; SI != SE; SI++)
			graph[temp->NodeNum][SI->getSUnit()->NodeNum] = 1;
	}

/*	dbgs() << "Initial Graph\n";
	for (i = 0; i < NumberOfSUnits; i++)
		dbgs() << i << "\t";
	dbgs() << "\n";
	for (i = 0; i < NumberOfSUnits; i++) {
		dbgs() << i << " : ";
		for (j = 0; j < NumberOfSUnits; j++)
			dbgs() << graph[i][j] << "\t";
		dbgs() << "\n";
	}

	dbgs() << "\n\n";
*/
	for (k = 0; k < NumberOfSUnits; k++)
		for (i = 0; i < NumberOfSUnits; i++)
			for (j = 0; j < NumberOfSUnits; j++)
				graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
	
	dbgs() << "All pair paths\n\t";
	for (i = 0; i < NumberOfSUnits; i++)
		dbgs() << i << "\t";
	dbgs() << "\n";
	for (i = 0; i < NumberOfSUnits; i++) {
		dbgs() << i << " : ";
		for (j = 0; j < NumberOfSUnits; j++)
			dbgs() << graph[i][j] << "\t";
		dbgs() << "\n";
	}

	int **LineageInterference;
	Lineage *L1, *L2;
	LineageInterference = (int **) calloc(NumberOfLineages, sizeof(int *));
	for (i = 0; i < NumberOfLineages; i++)
		LineageInterference[i] = (int *) calloc(NumberOfLineages, sizeof(int));

	for (i = 0; i < ListOfLineage.size(); i++) {
		L1 = ListOfLineage[i];
		for (j = i + 1; j < ListOfLineage.size(); j++) {

			/*if (i == j)
				break;
			*/
			L2 = ListOfLineage[j];
			if (graph[L1->StartNode->NodeNum][L2->EndNode->NodeNum] && graph[L2->StartNode->NodeNum][L1->EndNode->NodeNum]) {
				LineageInterference[L1->Id][L2->Id] = 1;
				LineageInterference[L2->Id][L1->Id] = 1;
				L1->Neighbours.push_back(L2);
				L1->NeighbourCount++;
				L2->Neighbours.push_back(L1);
				L2->NeighbourCount++;
				dbgs() << "Lineage: " << L1->Id << " and lineage " << L2->Id << " overlap\n";
			}
			if (graph[L1->StartNode->NodeNum][L2->EndNode->NodeNum] && (graph[L2->StartNode->NodeNum][L1->EndNode->NodeNum] == 0)) {
				dbgs() << "Fusing Lineage: " << L1->Id << "and Lineage: " << L2->Id << "\n";

				for (Lineage::Nodes_iterator LI = L2->Nodes.begin(), LE = L2->Nodes.end(); LI != LE; LI++) {
					(*LI)->lineage = L1;
					L1->setNode(*(LI));
				}
				for (Lineage::Neighbours_iterator NI = L2->Neighbours.begin(), NE = L2->Neighbours.end(); NI != NE; NI++) {

					bool NeighbourAlreadyPresent = false;
					for (Lineage::Neighbours_iterator NNI = L1->Neighbours.begin(), NNE = L1->Neighbours.end(); NNI != NNE; NNI++) {
						if (*NI == *NNI) {
							NeighbourAlreadyPresent = true;
							dbgs() << "Neighbour Lineage " << (*NI)->Id << " already present in Lineage " << L1->Id << "\n";
							break;
						}
					}
					if (!NeighbourAlreadyPresent) {

						L1->Neighbours.push_back(*NI);
						L1->NeighbourCount++;
						(*NI)->Neighbours.push_back(L1);
						(*NI)->NeighbourCount++;
						dbgs() << "Putting Lineage " << L1->Id << " into Neighbour Lineage " << (*NI)->Id << "\n";
					}
					for (Lineage::Neighbours_iterator NNI = (*NI)->Neighbours.begin(), NNE = (*NI)->Neighbours.end(); NNI != NNE; NNI++) {
					
						if (*NNI == L2)	{
	
							(*NI)->Neighbours.erase(NNI);
							dbgs() << "Erasing Lineage " << L2->Id << " from Neighbour Lineage " << (*NI)->Id << "\n";
						}
					}
					(*NI)->NeighbourCount--;
				}
				SDep SequencingEdge(L1->EndNode, SDep::Order);
				L2->StartNode->addPred(SequencingEdge);

				for (k = 0; k < NumberOfSUnits; k++) {
					if (graph[k][L1->EndNode->NodeNum]) {
						for (unsigned l = 0; l < NumberOfSUnits; l++) {
							if (graph[L2->StartNode->NodeNum][l])
								graph[k][l] = 1;
						}
					}
				}
				for (k = 0; k < NumberOfLineages; k++) {

					if (LineageInterference[L2->Id][k]) {
						LineageInterference[L2->Id][k] = -1;
						LineageInterference[k][L2->Id] = -1;
						LineageInterference[L1->Id][k] = 1;
						LineageInterference[k][L1->Id] = 1;
					}
				}
				L1->EndNode = L2->EndNode;
				ListOfLineage.erase(ListOfLineage.begin() + j);
				dbgs() << "Deleting Lineage " << L2->Id << "\n";
				dbgs() << "\n\nList of Lineages after deleting a Lineage\n";
				for (unsigned m = 0; m < ListOfLineage.size(); m++) {

					Lineage *L = ListOfLineage[m];
					dbgs() << "Lineage: " << L->Id << "StartNode: " << L->StartNode->NodeNum << ", EndNode: " << L->EndNode->NodeNum << ", List of Nodes: ";
					for (Lineage::Nodes_iterator LI = L->Nodes.begin(), LE = L->Nodes.end(); LI != LE; LI++) {
	
						dbgs() << (*LI)->NodeNum << " ";
					}
					dbgs() << ", List of Neighbours: ";
					for (Lineage::Neighbours_iterator NI = L->Neighbours.begin(), NE = L->Neighbours.end(); NI != NE; NI++) {
						dbgs() << (*NI)->Id << " ";
					}
					dbgs() << "\n";
				}
				dbgs() << "\n\n";
				j--;
			}
			if (graph[L2->StartNode->NodeNum][L1->EndNode->NodeNum] && (graph[L1->StartNode->NodeNum][L2->EndNode->NodeNum] == 0)) {
				dbgs() << "2nd Time Fusing Lineage: " << L2->Id << "and Lineage: " << L1->Id << "\n";

				for (Lineage::Nodes_iterator LI = L1->Nodes.begin(), LE = L1->Nodes.end(); LI != LE; LI++) {
					(*LI)->lineage = L2;
					L2->setNode(*(LI));
				}
				for (Lineage::Neighbours_iterator NI = L1->Neighbours.begin(), NE = L1->Neighbours.end(); NI != NE; NI++) {

					bool NeighbourAlreadyPresent = false;
					for (Lineage::Neighbours_iterator NNI = L2->Neighbours.begin(), NNE = L2->Neighbours.end(); NNI != NNE; NNI++) {
						if (*NI == *NNI) {
							NeighbourAlreadyPresent = true;
							dbgs() << "Neighbour Lineage " << (*NI)->Id << " already present in Lineage " << L2->Id << "\n";
							break;
						}
					}
					if (!NeighbourAlreadyPresent) {

						L2->Neighbours.push_back(*NI);
						L2->NeighbourCount++;
						(*NI)->Neighbours.push_back(L2);
						(*NI)->NeighbourCount++;
						dbgs() << "Putting Lineage " << L1->Id << " into Neighbour Lineage " << (*NI)->Id << "\n";
					}
					for (Lineage::Neighbours_iterator NNI = (*NI)->Neighbours.begin(), NNE = (*NI)->Neighbours.end(); NNI != NNE; NNI++) {
					
						if (*NNI == L1)	{
	
							(*NI)->Neighbours.erase(NNI);
							dbgs() << "Erasing Lineage " << L1->Id << " from Neighbour Lineage " << (*NI)->Id << "\n";
						}
					}
					(*NI)->NeighbourCount--;
				}
				SDep SequencingEdge(L2->EndNode, SDep::Order);
				L1->StartNode->addPred(SequencingEdge);

				for (k = 0; k < NumberOfSUnits; k++) {
					if (graph[k][L2->EndNode->NodeNum]) {
						for (unsigned l = 0; l < NumberOfSUnits; l++) {
							if (graph[L1->StartNode->NodeNum][l])
								graph[k][l] = 1;
						}
					}
				}
				for (k = 0; k < NumberOfLineages; k++) {

					if (LineageInterference[L1->Id][k]) {
						LineageInterference[L1->Id][k] = -1;
						LineageInterference[k][L1->Id] = -1;
						LineageInterference[L2->Id][k] = 1;
						LineageInterference[k][L2->Id] = 1;
					}
				}
				L2->EndNode = L1->EndNode;
				ListOfLineage.erase(ListOfLineage.begin() + i);
				dbgs() << "Deleting Lineage " << L1->Id << "\n";
				dbgs() << "\n\nList of Lineages after deleting a Lineage\n";
				for (unsigned m = 0; m < ListOfLineage.size(); m++) {

					Lineage *L = ListOfLineage[m];
					dbgs() << "Lineage: " << L->Id << "StartNode: " << L->StartNode->NodeNum << ", EndNode: " << L->EndNode->NodeNum << ", List of Nodes: ";
					for (Lineage::Nodes_iterator LI = L->Nodes.begin(), LE = L->Nodes.end(); LI != LE; LI++) {
	
						dbgs() << (*LI)->NodeNum << " ";
					}
					dbgs() << ", List of Neighbours: ";
					for (Lineage::Neighbours_iterator NI = L->Neighbours.begin(), NE = L->Neighbours.end(); NI != NE; NI++) {
						dbgs() << (*NI)->Id << " ";
					}
					dbgs() << "\n";
				}
				dbgs() << "\n\n";
				i--;
				break;
			}
		}
	}
	//NumberOfLineages = ListOfLineage.size();
	dbgs() << "Lineage Interference Graph\n\t";
	for (i = 0; i < ListOfLineage.size(); i++) {
		L1 = ListOfLineage[i];
		dbgs() << L1->Id << "\t";
	}
	dbgs() << "\n";
	for (i = 0; i < ListOfLineage.size(); i++) {
		L1 = ListOfLineage[i];
		dbgs() << L1->Id << ":" << L1->NeighbourCount << " : ";
		for (j = 0; j < ListOfLineage.size(); j++) {
			L2 = ListOfLineage[j];
			dbgs() << LineageInterference[L1->Id][L2->Id] << "\t";
		}
		dbgs() << "\n";
	}


	std::stack<Lineage *> StackOfLineages;
	std::vector<int> NeighboursLeft;
	std::vector<bool> AllocatedRegister;
	std::bitset<32> Conflict;

	NeighboursLeft.resize(NumberOfLineages);
	AllocatedRegister.resize(NumberOfLineages);
	
	for (i = 0; i < ListOfLineage.size(); i++) {
		L1 = ListOfLineage[i];
		NeighboursLeft[L1->Id] = L1->NeighbourCount;
	}

	for (i = 0; i < ListOfLineage.size(); i++) {
		L1 = ListOfLineage[i];
		if (NeighboursLeft[L1->Id] < 2) {
			StackOfLineages.push(L1);
			for (Lineage::Neighbours_iterator NI = L1->Neighbours.begin(), NE = L1->Neighbours.end(); NI != NE; NI++) {

				NeighboursLeft[(*NI)->Id]--;
			}
			NeighboursLeft[L1->Id] = -1;
		}
	}
	for (i = 0; i < ListOfLineage.size(); i++) {
		L1 = ListOfLineage[i];
		if (NeighboursLeft[L1->Id] >= 0) {
			StackOfLineages.push(L1);
		}
	}

	while (!StackOfLineages.empty()) {

		L1 = StackOfLineages.top();
		StackOfLineages.pop();
		Conflict.reset();
		for (Lineage::Neighbours_iterator NI = L1->Neighbours.begin(), NE = L1->Neighbours.end(); NI != NE; NI++) {
			if (AllocatedRegister[(*NI)->Id])
				Conflict[(*NI)->Reg] = 1;
		}
		j = 0;
		while(j <= 31 && Conflict[j]) {
			j++;
		}
		if (j > 32) {

			dbgs() << "Couldnt allocate register." << j << "\n";
			return;
		}
		dbgs() << "Register Allocated to Lineage " << L1->Id << " = " << j << "\n";
		AllocatedRegister[L1->Id] = 1;
		L1->Reg = j;
	}
	for (i = 0; i < ListOfLineage.size(); i++) {
		L1 = ListOfLineage[i];
		dbgs() << "Lineage " << L1->Id << " Register Allocated: " << L1->Reg << "\n";
	}

	ComputeHeight(Height);

	std::vector<SUnit *> ReadyList;
	std::vector<bool> RegAvailable;
	std::vector<SUnit *> LineageStartNodes;
	std::vector<SUnit *> LineageEndNodes;
	std::vector<int> SUnitPredsLeft;
	bool IsStartNode, IsEndNode;

	SUnitPredsLeft.resize(NumberOfSUnits);
	for (i = 0; i < ListOfLineage.size(); i++) {
		L1 = ListOfLineage[i];
		LineageStartNodes.push_back(L1->StartNode);
		LineageEndNodes.push_back(L1->EndNode);
	}

	for (i = 0; i < NumberOfSUnits; i++) {
	
		SUnitPredsLeft[i] = SUnits[i].Preds.size();
		if (SUnits[i].Preds.size() == 0) {

			ReadyList.push_back(&SUnits[i]);
			dbgs() << "Found the starting Node" << SUnits[i].NodeNum << "\n";
		}
	}

	int iteration = 0;
	Conflict.reset();
	while(!HeightOfSUnits.empty())
		HeightOfSUnits.pop();
	while(!ReadyList.empty()) {

		iteration++;
		dbgs() << "Iteration: " << iteration << "Ready List: ";

		for (std::vector<SUnit *>::iterator RI = ReadyList.begin(), RE = ReadyList.end(); RI != RE; RI++) {
			dbgs() << (*RI)->NodeNum << ", ";
			HeightOfSUnits.push(*RI);
		}
		dbgs() << "\n";

		while (!HeightOfSUnits.empty()) {
	
			IsStartNode = false, IsEndNode = false;
			temp = HeightOfSUnits.top();
			HeightOfSUnits.pop();
			if (temp->lineage == NULL) {
				dbgs() << "\tListing Control dependent node: " << temp->NodeNum << "\n";
				for (std::vector<SUnit *>::iterator RI = ReadyList.begin(), RE = ReadyList.end(); RI != RE; RI++) {

					if (*RI == temp)
						ReadyList.erase(RI);
				}
				for (SUnit::const_succ_iterator SI = temp->Succs.begin(), SE = temp->Succs.end(); SI != SE; SI++) {
				
					SUnitPredsLeft[SI->getSUnit()->NodeNum]--;
					if (SUnitPredsLeft[SI->getSUnit()->NodeNum] == 0) {
						dbgs() << "\t\tPushing Node to ReadyList: " << SI->getSUnit()->NodeNum;
						ReadyList.push_back(SI->getSUnit());
					}
				}
				continue;
			}
			for (std::vector<SUnit *>::iterator LI = LineageStartNodes.begin(), LE = LineageStartNodes.end(); LI != LE; LI++) {

				if (temp == *LI) {
					dbgs() << "Found " << temp->NodeNum << "Start Node in Lineage: " << (*LI)->lineage->Id << "\n";
					IsStartNode = true;
				}
			}
			if (!IsStartNode || (Conflict[temp->lineage->Reg] == 0)) {

				Conflict[temp->lineage->Reg] = 1;

				for (std::vector<SUnit *>::iterator RI = ReadyList.begin(), RE = ReadyList.end(); RI != RE; RI++) {

					if (*RI == temp) {
						ReadyList.erase(RI);
						break;
					}
				}
				dbgs() << "\tListing Node " << temp->NodeNum << "\n";
				for (SUnit::const_succ_iterator SI = temp->Succs.begin(), SE = temp->Succs.end(); SI != SE; SI++) {
				
					SUnitPredsLeft[SI->getSUnit()->NodeNum]--;
					if (SUnitPredsLeft[SI->getSUnit()->NodeNum] == 0) {
						dbgs() << "\t\tPushing Node to ReadyList: " << SI->getSUnit()->NodeNum;
						ReadyList.push_back(SI->getSUnit());
					}
				}
				for (std::vector<SUnit *>::iterator LI = LineageEndNodes.begin(), LE = LineageEndNodes.end(); LI != LE; LI++) {
	
					if (temp == *LI) {
						dbgs() << "Found " << temp->NodeNum << "End Node in Lineage: " << (*LI)->lineage->Id << "\n";
						IsEndNode = true;
					}
				}
				if (IsEndNode) {
					
					for (unsigned l = 0; l < ListOfLineage.size(); l++) {

						Lineage *temp_lineage = ListOfLineage[l];
						if (temp_lineage->EndNode == temp) {
							
							Conflict[temp_lineage->Reg] = 0;
							dbgs() << "\t\tReleasing Register " << temp_lineage->Reg << " from Node \n" << temp->NodeNum;
						}
					}
				}
			}
		}
	}
}

/// BuildSchedGraph - Build the SUnit graph from the selection dag that we
/// are input.  This SUnit graph is similar to the SelectionDAG, but
/// excludes nodes that aren't interesting to scheduling, and represents
/// glued together nodes with a single SUnit.
void ScheduleDAGSDNodes::BuildSchedGraph(AliasAnalysis *AA) {
  // Cluster certain nodes which should be scheduled together.
  ClusterNodes();
  // Populate the SUnits array.
  BuildSchedUnits();
  // Compute all the scheduling dependencies between nodes.
  AddSchedEdges();

  this->viewGraph();
  // The MRIS implementation
  BuildDDG();

}

// Initialize NumNodeDefs for the current Node's opcode.
void ScheduleDAGSDNodes::RegDefIter::InitNodeNumDefs() {
  // Check for phys reg copy.
  if (!Node)
    return;

  if (!Node->isMachineOpcode()) {
    if (Node->getOpcode() == ISD::CopyFromReg)
      NodeNumDefs = 1;
    else
      NodeNumDefs = 0;
    return;
  }
  unsigned POpc = Node->getMachineOpcode();
  if (POpc == TargetOpcode::IMPLICIT_DEF) {
    // No register need be allocated for this.
    NodeNumDefs = 0;
    return;
  }
  unsigned NRegDefs = SchedDAG->TII->get(Node->getMachineOpcode()).getNumDefs();
  // Some instructions define regs that are not represented in the selection DAG
  // (e.g. unused flags). See tMOVi8. Make sure we don't access past NumValues.
  NodeNumDefs = std::min(Node->getNumValues(), NRegDefs);
  DefIdx = 0;
}

// Construct a RegDefIter for this SUnit and find the first valid value.
ScheduleDAGSDNodes::RegDefIter::RegDefIter(const SUnit *SU,
                                           const ScheduleDAGSDNodes *SD)
  : SchedDAG(SD), Node(SU->getNode()), DefIdx(0), NodeNumDefs(0) {
  InitNodeNumDefs();
  Advance();
}

// Advance to the next valid value defined by the SUnit.
void ScheduleDAGSDNodes::RegDefIter::Advance() {
  for (;Node;) { // Visit all glued nodes.
    for (;DefIdx < NodeNumDefs; ++DefIdx) {
      if (!Node->hasAnyUseOfValue(DefIdx))
        continue;
      ValueType = Node->getValueType(DefIdx);
      ++DefIdx;
      return; // Found a normal regdef.
    }
    Node = Node->getGluedNode();
    if (Node == NULL) {
      return; // No values left to visit.
    }
    InitNodeNumDefs();
  }
}

void ScheduleDAGSDNodes::InitNumRegDefsLeft(SUnit *SU) {
  assert(SU->NumRegDefsLeft == 0 && "expect a new node");
  for (RegDefIter I(SU, this); I.IsValid(); I.Advance()) {
    assert(SU->NumRegDefsLeft < USHRT_MAX && "overflow is ok but unexpected");
    ++SU->NumRegDefsLeft;
  }
}

void ScheduleDAGSDNodes::ComputeLatency(SUnit *SU) {
  SDNode *N = SU->getNode();

  // TokenFactor operands are considered zero latency, and some schedulers
  // (e.g. Top-Down list) may rely on the fact that operand latency is nonzero
  // whenever node latency is nonzero.
  if (N && N->getOpcode() == ISD::TokenFactor) {
    SU->Latency = 0;
    return;
  }

  // Check to see if the scheduler cares about latencies.
  if (ForceUnitLatencies()) {
    SU->Latency = 1;
    return;
  }

  if (!InstrItins || InstrItins->isEmpty()) {
    if (N && N->isMachineOpcode() &&
        TII->isHighLatencyDef(N->getMachineOpcode()))
      SU->Latency = HighLatencyCycles;
    else
      SU->Latency = 1;
    return;
  }

  // Compute the latency for the node.  We use the sum of the latencies for
  // all nodes glued together into this SUnit.
  SU->Latency = 0;
  for (SDNode *N = SU->getNode(); N; N = N->getGluedNode())
    if (N->isMachineOpcode())
      SU->Latency += TII->getInstrLatency(InstrItins, N);
}

void ScheduleDAGSDNodes::ComputeOperandLatency(SDNode *Def, SDNode *Use,
                                               unsigned OpIdx, SDep& dep) const{
  // Check to see if the scheduler cares about latencies.
  if (ForceUnitLatencies())
    return;

  if (dep.getKind() != SDep::Data)
    return;

  unsigned DefIdx = Use->getOperand(OpIdx).getResNo();
  if (Use->isMachineOpcode())
    // Adjust the use operand index by num of defs.
    OpIdx += TII->get(Use->getMachineOpcode()).getNumDefs();
  int Latency = TII->getOperandLatency(InstrItins, Def, DefIdx, Use, OpIdx);
  if (Latency > 1 && Use->getOpcode() == ISD::CopyToReg &&
      !BB->succ_empty()) {
    unsigned Reg = cast<RegisterSDNode>(Use->getOperand(1))->getReg();
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      // This copy is a liveout value. It is likely coalesced, so reduce the
      // latency so not to penalize the def.
      // FIXME: need target specific adjustment here?
      Latency = (Latency > 1) ? Latency - 1 : 1;
  }
  if (Latency >= 0)
    dep.setLatency(Latency);
}

void ScheduleDAGSDNodes::dumpNode(const SUnit *SU) const {
  if (!SU->getNode()) {
    dbgs() << "PHYS REG COPY\n";
    return;
  }

  SU->getNode()->dump(DAG);
  dbgs() << "\n";
  SmallVector<SDNode *, 4> GluedNodes;
  for (SDNode *N = SU->getNode()->getGluedNode(); N; N = N->getGluedNode())
    GluedNodes.push_back(N);
  while (!GluedNodes.empty()) {
    dbgs() << "    ";
    GluedNodes.back()->dump(DAG);
    dbgs() << "\n";
    GluedNodes.pop_back();
  }
}

namespace {
  struct OrderSorter {
    bool operator()(const std::pair<unsigned, MachineInstr*> &A,
                    const std::pair<unsigned, MachineInstr*> &B) {
      return A.first < B.first;
    }
  };
}

/// ProcessSDDbgValues - Process SDDbgValues associated with this node.
static void ProcessSDDbgValues(SDNode *N, SelectionDAG *DAG,
                               InstrEmitter &Emitter,
                    SmallVector<std::pair<unsigned, MachineInstr*>, 32> &Orders,
                            DenseMap<SDValue, unsigned> &VRBaseMap,
                            unsigned Order) {
  if (!N->getHasDebugValue())
    return;

  // Opportunistically insert immediate dbg_value uses, i.e. those with source
  // order number right after the N.
  MachineBasicBlock *BB = Emitter.getBlock();
  MachineBasicBlock::iterator InsertPos = Emitter.getInsertPos();
  ArrayRef<SDDbgValue*> DVs = DAG->GetDbgValues(N);
  for (unsigned i = 0, e = DVs.size(); i != e; ++i) {
    if (DVs[i]->isInvalidated())
      continue;
    unsigned DVOrder = DVs[i]->getOrder();
    if (!Order || DVOrder == ++Order) {
      MachineInstr *DbgMI = Emitter.EmitDbgValue(DVs[i], VRBaseMap);
      if (DbgMI) {
        Orders.push_back(std::make_pair(DVOrder, DbgMI));
        BB->insert(InsertPos, DbgMI);
      }
      DVs[i]->setIsInvalidated();
    }
  }
}

// ProcessSourceNode - Process nodes with source order numbers. These are added
// to a vector which EmitSchedule uses to determine how to insert dbg_value
// instructions in the right order.
static void ProcessSourceNode(SDNode *N, SelectionDAG *DAG,
                           InstrEmitter &Emitter,
                           DenseMap<SDValue, unsigned> &VRBaseMap,
                    SmallVector<std::pair<unsigned, MachineInstr*>, 32> &Orders,
                           SmallSet<unsigned, 8> &Seen) {
  unsigned Order = DAG->GetOrdering(N);
  if (!Order || !Seen.insert(Order)) {
    // Process any valid SDDbgValues even if node does not have any order
    // assigned.
    ProcessSDDbgValues(N, DAG, Emitter, Orders, VRBaseMap, 0);
    return;
  }

  MachineBasicBlock *BB = Emitter.getBlock();
  if (Emitter.getInsertPos() == BB->begin() || BB->back().isPHI()) {
    // Did not insert any instruction.
    Orders.push_back(std::make_pair(Order, (MachineInstr*)0));
    return;
  }

  Orders.push_back(std::make_pair(Order, prior(Emitter.getInsertPos())));
  ProcessSDDbgValues(N, DAG, Emitter, Orders, VRBaseMap, Order);
}


/// EmitSchedule - Emit the machine code in scheduled order.
MachineBasicBlock *ScheduleDAGSDNodes::EmitSchedule() {
  InstrEmitter Emitter(BB, InsertPos);
  DenseMap<SDValue, unsigned> VRBaseMap;
  DenseMap<SUnit*, unsigned> CopyVRBaseMap;
  SmallVector<std::pair<unsigned, MachineInstr*>, 32> Orders;
  SmallSet<unsigned, 8> Seen;
  bool HasDbg = DAG->hasDebugValues();

  // If this is the first BB, emit byval parameter dbg_value's.
  if (HasDbg && BB->getParent()->begin() == MachineFunction::iterator(BB)) {
    SDDbgInfo::DbgIterator PDI = DAG->ByvalParmDbgBegin();
    SDDbgInfo::DbgIterator PDE = DAG->ByvalParmDbgEnd();
    for (; PDI != PDE; ++PDI) {
      MachineInstr *DbgMI= Emitter.EmitDbgValue(*PDI, VRBaseMap);
      if (DbgMI)
        BB->insert(InsertPos, DbgMI);
    }
  }

  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    SUnit *SU = Sequence[i];
    if (!SU) {
      // Null SUnit* is a noop.
      EmitNoop();
      continue;
    }

    // For pre-regalloc scheduling, create instructions corresponding to the
    // SDNode and any glued SDNodes and append them to the block.
    if (!SU->getNode()) {
      // Emit a copy.
      EmitPhysRegCopy(SU, CopyVRBaseMap);
      continue;
    }

    SmallVector<SDNode *, 4> GluedNodes;
    for (SDNode *N = SU->getNode()->getGluedNode(); N;
         N = N->getGluedNode())
      GluedNodes.push_back(N);
    while (!GluedNodes.empty()) {
      SDNode *N = GluedNodes.back();
      Emitter.EmitNode(GluedNodes.back(), SU->OrigNode != SU, SU->isCloned,
                       VRBaseMap);
      // Remember the source order of the inserted instruction.
      if (HasDbg)
        ProcessSourceNode(N, DAG, Emitter, VRBaseMap, Orders, Seen);
      GluedNodes.pop_back();
    }
    Emitter.EmitNode(SU->getNode(), SU->OrigNode != SU, SU->isCloned,
                     VRBaseMap);
    // Remember the source order of the inserted instruction.
    if (HasDbg)
      ProcessSourceNode(SU->getNode(), DAG, Emitter, VRBaseMap, Orders,
                        Seen);
  }

  // Insert all the dbg_values which have not already been inserted in source
  // order sequence.
  if (HasDbg) {
    MachineBasicBlock::iterator BBBegin = BB->getFirstNonPHI();

    // Sort the source order instructions and use the order to insert debug
    // values.
    std::sort(Orders.begin(), Orders.end(), OrderSorter());

    SDDbgInfo::DbgIterator DI = DAG->DbgBegin();
    SDDbgInfo::DbgIterator DE = DAG->DbgEnd();
    // Now emit the rest according to source order.
    unsigned LastOrder = 0;
    for (unsigned i = 0, e = Orders.size(); i != e && DI != DE; ++i) {
      unsigned Order = Orders[i].first;
      MachineInstr *MI = Orders[i].second;
      // Insert all SDDbgValue's whose order(s) are before "Order".
      if (!MI)
        continue;
      for (; DI != DE &&
             (*DI)->getOrder() >= LastOrder && (*DI)->getOrder() < Order; ++DI) {
        if ((*DI)->isInvalidated())
          continue;
        MachineInstr *DbgMI = Emitter.EmitDbgValue(*DI, VRBaseMap);
        if (DbgMI) {
          if (!LastOrder)
            // Insert to start of the BB (after PHIs).
            BB->insert(BBBegin, DbgMI);
          else {
            // Insert at the instruction, which may be in a different
            // block, if the block was split by a custom inserter.
            MachineBasicBlock::iterator Pos = MI;
            MI->getParent()->insert(llvm::next(Pos), DbgMI);
          }
        }
      }
      LastOrder = Order;
    }
    // Add trailing DbgValue's before the terminator. FIXME: May want to add
    // some of them before one or more conditional branches?
    while (DI != DE) {
      MachineBasicBlock *InsertBB = Emitter.getBlock();
      MachineBasicBlock::iterator Pos= Emitter.getBlock()->getFirstTerminator();
      if (!(*DI)->isInvalidated()) {
        MachineInstr *DbgMI= Emitter.EmitDbgValue(*DI, VRBaseMap);
        if (DbgMI)
          InsertBB->insert(Pos, DbgMI);
      }
      ++DI;
    }
  }

  BB = Emitter.getBlock();
  InsertPos = Emitter.getInsertPos();
  return BB;
}
